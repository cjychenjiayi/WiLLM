% 对每组载波进行相位校正
% You are facing the Mona Lisa Spot Localization using PHY Layer Information
% function phase = calibrate_phase(phase_origin)
% [PA,TX,RX,SC] = size(phase_origin);
% phase = zeros(PA,TX,RX,SC);
% 
% SubCarrInd = [-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,-1,1,3,5,7,9,11,13,15,17,19,21,23,25,27,28];
% 
% for tx = 1:TX
%     for rx = 1:RX
%         for sc = 1:SC
%             fiber = squeeze(phase_origin(:,tx,rx,sc));
%             diff = 0;
%         for pa = 2:PA
% 
%             plot(fiber)
%             相位解缠
% 
%                 tmp = fiber(pa)-fiber(pa-1);
%                 if abs(tmp) > pi
%                     diff = diff + 1*sign(tmp);
%                 end
%                 fiber(pa) = fiber(pa) - diff*2*pi;
%                 phase(pa,tx,rx,sc) = fiber(pa);
%             end
%             plot(fiber)
% 
%             线性校正
%             a = (fiber(SC)-fiber(1))/(SC-1);
%             b = mean(fiber);
%             for sc = 1:SC
%                 phase(tx,rx,sc,pa) = fiber(sc)-a*sc-b;
%             end
%             a = (fiber(SC)-fiber(1))/(SubCarrInd(SC)- SubCarrInd(1));
%             b = mean(fiber);
%             for sc = 1:SC
%                 phase(pa,tx,rx,sc) = fiber(sc)-a*SubCarrInd(sc)-b;
%             end
% 
%             plot(squeeze(phase(tx,rx,:,pa)))
%         end
%     end
% end

% end

function phase = calibrate_phase(phase_origin)
[PA,TX,RX,SC] = size(phase_origin);
phase = zeros(PA,TX,RX,SC);

SubCarrInd = [-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,-1,1,3,5,7,9,11,13,15,17,19,21,23,25,27,28];

for tx = 1:TX
    for rx = 1:RX
        for pa = 1:PA
            fiber = squeeze(phase_origin(pa,tx,rx,:));
            diff = 0;
%             plot(fiber)
            % 相位解缠
            phase(pa,tx,rx,1) = fiber(1);
            for sc = 2:SC
                tmp = fiber(sc)-fiber(sc-1);
                if abs(tmp) > pi
                    diff = diff + 1*sign(tmp);
                end
                phase(pa,tx,rx,sc) = fiber(sc) - diff*2*pi;
            end
%             plot(fiber)

            % 线性校正
            % a = (fiber(SC)-fiber(1))/(SC-1);
            % b = mean(fiber);
            % for sc = 1:SC
            %     phase(tx,rx,sc,pa) = fiber(sc)-a*sc-b;
            % end
            fiber = squeeze(phase(pa,tx,rx,:));
            a = (fiber(SC)-fiber(1))/(SubCarrInd(SC)- SubCarrInd(1));
            b = mean(fiber);
            for sc = 1:SC
                phase(pa,tx,rx,sc) = fiber(sc)-a*SubCarrInd(sc)-b;
            end

            % plot(squeeze(phase(tx,rx,:,pa)))
        end
    end
end

end
